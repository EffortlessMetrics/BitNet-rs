//! Model deployment lifecycle manager for GPU inference.
//!
//! Provides blue/green, canary, and rolling deployment strategies with
//! health checks, automatic rollback, and deployment metrics tracking.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Top-level deployment configuration.
#[derive(Debug, Clone)]
pub struct DeploymentConfig {
    /// Maximum number of models that can be loaded simultaneously.
    pub max_models: usize,
    /// Interval between health checks in milliseconds.
    pub health_check_interval_ms: u64,
    /// Whether to automatically rollback on deployment failure.
    pub rollback_on_failure: bool,
    /// Percentage of traffic for canary deployments (0.0–100.0).
    pub canary_percentage: f32,
    /// Number of warmup requests before marking a deployment active.
    pub warmup_requests: u32,
}

impl Default for DeploymentConfig {
    fn default() -> Self {
        Self {
            max_models: 4,
            health_check_interval_ms: 5000,
            rollback_on_failure: true,
            canary_percentage: 10.0,
            warmup_requests: 5,
        }
    }
}

impl DeploymentConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_models == 0 {
            return Err("max_models must be > 0".into());
        }
        if self.health_check_interval_ms == 0 {
            return Err("health_check_interval_ms must be > 0".into());
        }
        if !(0.0..=100.0).contains(&self.canary_percentage) {
            return Err("canary_percentage must be in [0.0, 100.0]".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Model version
// ---------------------------------------------------------------------------

/// A versioned model artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelVersion {
    pub id: String,
    pub version: u64,
    pub path: String,
    pub size_bytes: u64,
    pub created_at: u64,
    pub status: ModelStatus,
}

/// Status of a model artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    Available,
    Loading,
    Loaded,
    Unloading,
    Failed,
}

// ---------------------------------------------------------------------------
// Deployment status
// ---------------------------------------------------------------------------

/// Lifecycle status of a single deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeploymentStatus {
    Pending,
    Deploying,
    Warming,
    Active,
    Draining,
    RolledBack,
    Failed,
}

impl DeploymentStatus {
    /// Returns `true` when the deployment is in a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Active | Self::RolledBack | Self::Failed)
    }
}

// ---------------------------------------------------------------------------
// Deployment slot
// ---------------------------------------------------------------------------

/// A deployment slot that holds a single model version.
#[derive(Debug, Clone)]
pub struct DeploymentSlot {
    pub slot_id: usize,
    pub model_version: Option<ModelVersion>,
    pub status: DeploymentStatus,
    pub traffic_weight: f32,
    pub health: HealthCheck,
}

impl DeploymentSlot {
    pub fn new(slot_id: usize) -> Self {
        Self {
            slot_id,
            model_version: None,
            status: DeploymentStatus::Pending,
            traffic_weight: 0.0,
            health: HealthCheck::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

/// Result of a health check against a deployment slot.
#[derive(Debug, Clone)]
pub struct HealthCheck {
    pub is_healthy: bool,
    pub latency_ms: f64,
    pub error_count: u32,
    pub last_check_at: Option<Instant>,
}

impl Default for HealthCheck {
    fn default() -> Self {
        Self { is_healthy: true, latency_ms: 0.0, error_count: 0, last_check_at: None }
    }
}

// ---------------------------------------------------------------------------
// Rollout strategy
// ---------------------------------------------------------------------------

/// Strategy used to roll out a new model version.
#[derive(Debug, Clone, PartialEq)]
pub enum RolloutStrategy {
    /// Swap two identical slot groups atomically.
    BlueGreen,
    /// Gradually shift traffic to the new version.
    Canary(f32),
    /// Replace slots one at a time.
    Rolling(usize),
    /// Replace immediately with no gradual shift.
    Immediate,
}

// ---------------------------------------------------------------------------
// Deployment plan / steps
// ---------------------------------------------------------------------------

/// High-level plan for a deployment.
#[derive(Debug, Clone)]
pub struct DeploymentPlan {
    pub source_version: Option<ModelVersion>,
    pub target_version: ModelVersion,
    pub strategy: RolloutStrategy,
    pub steps: Vec<DeploymentStep>,
}

/// Action inside a deployment step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeploymentAction {
    LoadModel,
    Warmup,
    ShiftTraffic,
    Validate,
    DrainOld,
    UnloadModel,
}

/// A single step within a deployment plan.
#[derive(Debug, Clone)]
pub struct DeploymentStep {
    pub action: DeploymentAction,
    pub slot_id: usize,
    pub traffic_shift: f32,
    pub validation_required: bool,
}

// ---------------------------------------------------------------------------
// Rollback policy
// ---------------------------------------------------------------------------

/// Policy controlling when automatic rollback triggers.
#[derive(Debug, Clone)]
pub struct RollbackPolicy {
    pub max_errors: u32,
    pub max_latency_ms: f64,
    pub auto_rollback: bool,
}

impl Default for RollbackPolicy {
    fn default() -> Self {
        Self { max_errors: 3, max_latency_ms: 1000.0, auto_rollback: true }
    }
}

// ---------------------------------------------------------------------------
// Deployment metrics
// ---------------------------------------------------------------------------

/// Aggregate metrics across deployments.
#[derive(Debug, Clone, Default)]
pub struct DeploymentMetrics {
    pub total_deployments: u64,
    pub successful_deployments: u64,
    pub rollbacks: u64,
    pub avg_deploy_time_ms: f64,
    pub deploy_times_ms: Vec<f64>,
    /// Cumulative uptime per model version string.
    pub uptime_per_version: HashMap<String, Duration>,
}

impl DeploymentMetrics {
    /// Record a completed deployment.
    pub fn record_deployment(&mut self, time_ms: f64, success: bool) {
        self.total_deployments += 1;
        if success {
            self.successful_deployments += 1;
        }
        self.deploy_times_ms.push(time_ms);
        self.avg_deploy_time_ms =
            self.deploy_times_ms.iter().sum::<f64>() / self.deploy_times_ms.len() as f64;
    }

    /// Record a rollback event.
    pub fn record_rollback(&mut self) {
        self.rollbacks += 1;
    }

    /// Add uptime for a specific version.
    pub fn add_uptime(&mut self, version_key: &str, duration: Duration) {
        let entry = self.uptime_per_version.entry(version_key.to_string()).or_default();
        *entry += duration;
    }
}

// ---------------------------------------------------------------------------
// DeploymentManager
// ---------------------------------------------------------------------------

/// Manages the lifecycle of model deployments across a set of slots.
#[derive(Debug)]
pub struct DeploymentManager {
    config: DeploymentConfig,
    slots: Vec<DeploymentSlot>,
    rollback_policy: RollbackPolicy,
    metrics: DeploymentMetrics,
    /// Stack of previously active model versions for rollback.
    version_history: Vec<ModelVersion>,
    /// Currently executing plan, if any.
    active_plan: Option<DeploymentPlan>,
}

impl DeploymentManager {
    /// Create a new manager with the given config.
    pub fn new(config: DeploymentConfig) -> Result<Self, String> {
        config.validate()?;
        let slots = (0..config.max_models).map(DeploymentSlot::new).collect();
        Ok(Self {
            config,
            slots,
            rollback_policy: RollbackPolicy::default(),
            metrics: DeploymentMetrics::default(),
            version_history: Vec::new(),
            active_plan: None,
        })
    }

    /// Create a manager with explicit rollback policy.
    pub fn with_rollback_policy(mut self, policy: RollbackPolicy) -> Self {
        self.rollback_policy = policy;
        self
    }

    // -- Accessors ----------------------------------------------------------

    pub fn config(&self) -> &DeploymentConfig {
        &self.config
    }

    pub fn slots(&self) -> &[DeploymentSlot] {
        &self.slots
    }

    pub fn metrics(&self) -> &DeploymentMetrics {
        &self.metrics
    }

    pub fn active_plan(&self) -> Option<&DeploymentPlan> {
        self.active_plan.as_ref()
    }

    pub fn version_history(&self) -> &[ModelVersion] {
        &self.version_history
    }

    /// Return the first slot that is `Active`.
    pub fn get_active_slot(&self) -> Option<&DeploymentSlot> {
        self.slots.iter().find(|s| s.status == DeploymentStatus::Active)
    }

    /// Return all slots that currently receive traffic.
    pub fn get_serving_slots(&self) -> Vec<&DeploymentSlot> {
        self.slots.iter().filter(|s| s.traffic_weight > 0.0).collect()
    }

    // -- Plan generation ----------------------------------------------------

    /// Build a deployment plan for the given strategy.
    pub fn plan_deployment(
        &self,
        target: ModelVersion,
        strategy: RolloutStrategy,
    ) -> Result<DeploymentPlan, String> {
        let source = self.get_active_slot().and_then(|s| s.model_version.clone());

        let steps = match &strategy {
            RolloutStrategy::BlueGreen => self.plan_blue_green(&target),
            RolloutStrategy::Canary(pct) => self.plan_canary(&target, *pct),
            RolloutStrategy::Rolling(batch) => self.plan_rolling(&target, *batch),
            RolloutStrategy::Immediate => self.plan_immediate(&target),
        };

        Ok(DeploymentPlan { source_version: source, target_version: target, strategy, steps })
    }

    fn plan_blue_green(&self, _target: &ModelVersion) -> Vec<DeploymentStep> {
        // Find an idle slot for the new version, then swap traffic.
        let idle_slot = self
            .slots
            .iter()
            .find(|s| s.status == DeploymentStatus::Pending)
            .map(|s| s.slot_id)
            .unwrap_or(1);

        vec![
            DeploymentStep {
                action: DeploymentAction::LoadModel,
                slot_id: idle_slot,
                traffic_shift: 0.0,
                validation_required: false,
            },
            DeploymentStep {
                action: DeploymentAction::Warmup,
                slot_id: idle_slot,
                traffic_shift: 0.0,
                validation_required: false,
            },
            DeploymentStep {
                action: DeploymentAction::Validate,
                slot_id: idle_slot,
                traffic_shift: 0.0,
                validation_required: true,
            },
            DeploymentStep {
                action: DeploymentAction::ShiftTraffic,
                slot_id: idle_slot,
                traffic_shift: 100.0,
                validation_required: false,
            },
            DeploymentStep {
                action: DeploymentAction::DrainOld,
                slot_id: 0,
                traffic_shift: 0.0,
                validation_required: false,
            },
        ]
    }

    fn plan_canary(&self, _target: &ModelVersion, pct: f32) -> Vec<DeploymentStep> {
        let idle_slot = self
            .slots
            .iter()
            .find(|s| s.status == DeploymentStatus::Pending)
            .map(|s| s.slot_id)
            .unwrap_or(1);

        let mut steps = vec![
            DeploymentStep {
                action: DeploymentAction::LoadModel,
                slot_id: idle_slot,
                traffic_shift: 0.0,
                validation_required: false,
            },
            DeploymentStep {
                action: DeploymentAction::Warmup,
                slot_id: idle_slot,
                traffic_shift: 0.0,
                validation_required: false,
            },
        ];

        // Gradually increase traffic.
        let mut current = pct;
        while current < 100.0 {
            steps.push(DeploymentStep {
                action: DeploymentAction::ShiftTraffic,
                slot_id: idle_slot,
                traffic_shift: current,
                validation_required: true,
            });
            current = (current * 2.0).min(100.0);
            if (current - 100.0).abs() < f32::EPSILON {
                break;
            }
        }

        steps.push(DeploymentStep {
            action: DeploymentAction::ShiftTraffic,
            slot_id: idle_slot,
            traffic_shift: 100.0,
            validation_required: true,
        });
        steps.push(DeploymentStep {
            action: DeploymentAction::DrainOld,
            slot_id: 0,
            traffic_shift: 0.0,
            validation_required: false,
        });

        steps
    }

    fn plan_rolling(&self, _target: &ModelVersion, batch_size: usize) -> Vec<DeploymentStep> {
        let mut steps = Vec::new();
        for chunk in self.slots.chunks(batch_size.max(1)) {
            for slot in chunk {
                steps.push(DeploymentStep {
                    action: DeploymentAction::LoadModel,
                    slot_id: slot.slot_id,
                    traffic_shift: 0.0,
                    validation_required: false,
                });
                steps.push(DeploymentStep {
                    action: DeploymentAction::Warmup,
                    slot_id: slot.slot_id,
                    traffic_shift: 0.0,
                    validation_required: false,
                });
                steps.push(DeploymentStep {
                    action: DeploymentAction::Validate,
                    slot_id: slot.slot_id,
                    traffic_shift: 100.0 / self.slots.len() as f32,
                    validation_required: true,
                });
            }
        }
        steps
    }

    fn plan_immediate(&self, _target: &ModelVersion) -> Vec<DeploymentStep> {
        let slot_id = self.get_active_slot().map(|s| s.slot_id).unwrap_or(0);
        vec![
            DeploymentStep {
                action: DeploymentAction::LoadModel,
                slot_id,
                traffic_shift: 0.0,
                validation_required: false,
            },
            DeploymentStep {
                action: DeploymentAction::ShiftTraffic,
                slot_id,
                traffic_shift: 100.0,
                validation_required: false,
            },
        ]
    }

    // -- Deployment execution -----------------------------------------------

    /// Begin deploying a model version according to the given strategy.
    pub fn deploy(
        &mut self,
        target: ModelVersion,
        strategy: RolloutStrategy,
    ) -> Result<DeploymentPlan, String> {
        if self.active_plan.is_some() {
            return Err("deployment already in progress".into());
        }

        let plan = self.plan_deployment(target.clone(), strategy)?;

        // Snapshot the current active version for rollback.
        if let Some(active) = self.get_active_slot().and_then(|s| s.model_version.clone()) {
            self.version_history.push(active);
        }

        self.active_plan = Some(plan.clone());
        Ok(plan)
    }

    /// Execute the next step in the active deployment plan.
    /// Returns the step that was executed, or `None` if the plan is complete.
    pub fn execute_next_step(&mut self) -> Result<Option<DeploymentStep>, String> {
        let (step, target, remaining) = {
            let plan = self.active_plan.as_mut().ok_or("no active deployment plan")?;
            if plan.steps.is_empty() {
                self.active_plan = None;
                return Ok(None);
            }
            let step = plan.steps.remove(0);
            let target = plan.target_version.clone();
            let remaining = plan.steps.len();
            (step, target, remaining)
        };

        self.apply_step(&step, &target)?;

        if remaining == 0 {
            self.finalize_deployment()?;
        }

        Ok(Some(step))
    }

    /// Execute all remaining steps.
    pub fn execute_all_steps(&mut self) -> Result<Vec<DeploymentStep>, String> {
        let mut executed = Vec::new();
        while self.active_plan.is_some() {
            match self.execute_next_step()? {
                Some(step) => executed.push(step),
                None => break,
            }
        }
        Ok(executed)
    }

    fn apply_step(&mut self, step: &DeploymentStep, target: &ModelVersion) -> Result<(), String> {
        let slot = self
            .slots
            .get_mut(step.slot_id)
            .ok_or_else(|| format!("slot {} not found", step.slot_id))?;

        match step.action {
            DeploymentAction::LoadModel => {
                slot.model_version = Some(target.clone());
                slot.status = DeploymentStatus::Deploying;
            }
            DeploymentAction::Warmup => {
                slot.status = DeploymentStatus::Warming;
            }
            DeploymentAction::ShiftTraffic => {
                slot.traffic_weight = step.traffic_shift;
                if (step.traffic_shift - 100.0).abs() < f32::EPSILON {
                    slot.status = DeploymentStatus::Active;
                }
                // Reduce traffic on other slots proportionally.
                let remaining = 100.0 - step.traffic_shift;
                let other_serving: Vec<usize> = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|(i, s)| *i != step.slot_id && s.traffic_weight > 0.0)
                    .map(|(i, _)| i)
                    .collect();
                let count = other_serving.len();
                for idx in other_serving {
                    self.slots[idx].traffic_weight =
                        if count > 0 { remaining / count as f32 } else { 0.0 };
                }
            }
            DeploymentAction::Validate => {
                if !slot.health.is_healthy {
                    return Err(format!("slot {} failed validation", step.slot_id));
                }
            }
            DeploymentAction::DrainOld => {
                slot.status = DeploymentStatus::Draining;
                slot.traffic_weight = 0.0;
            }
            DeploymentAction::UnloadModel => {
                slot.model_version = None;
                slot.status = DeploymentStatus::Pending;
                slot.traffic_weight = 0.0;
            }
        }
        Ok(())
    }

    fn finalize_deployment(&mut self) -> Result<(), String> {
        self.metrics.record_deployment(100.0, true);
        self.active_plan = None;
        Ok(())
    }

    // -- Rollback -----------------------------------------------------------

    /// Rollback to the previous model version.
    pub fn rollback(&mut self) -> Result<ModelVersion, String> {
        let prev = self.version_history.pop().ok_or("no previous version to rollback to")?;

        // Cancel any in-flight plan.
        self.active_plan = None;

        // Mark all non-pending slots as rolled-back and clear traffic.
        for slot in &mut self.slots {
            if slot.status != DeploymentStatus::Pending {
                slot.status = DeploymentStatus::RolledBack;
                slot.traffic_weight = 0.0;
            }
        }

        // Re-deploy the previous version on slot 0.
        let slot = &mut self.slots[0];
        slot.model_version = Some(prev.clone());
        slot.status = DeploymentStatus::Active;
        slot.traffic_weight = 100.0;

        self.metrics.record_rollback();
        self.metrics.record_deployment(50.0, false);

        Ok(prev)
    }

    // -- Draining -----------------------------------------------------------

    /// Drain a specific slot, shifting its traffic to remaining active slots.
    pub fn drain_slot(&mut self, slot_id: usize) -> Result<(), String> {
        if slot_id >= self.slots.len() {
            return Err(format!("slot {} out of range", slot_id));
        }

        let drained_weight = self.slots[slot_id].traffic_weight;
        self.slots[slot_id].status = DeploymentStatus::Draining;
        self.slots[slot_id].traffic_weight = 0.0;

        let active_slots: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(i, s)| {
                *i != slot_id && s.traffic_weight > 0.0 && s.status == DeploymentStatus::Active
            })
            .map(|(i, _)| i)
            .collect();

        if !active_slots.is_empty() {
            let share = drained_weight / active_slots.len() as f32;
            for idx in active_slots {
                self.slots[idx].traffic_weight += share;
            }
        }

        Ok(())
    }

    // -- Health checks ------------------------------------------------------

    /// Run a health check on the specified slot.
    pub fn health_check(
        &mut self,
        slot_id: usize,
        healthy: bool,
        latency_ms: f64,
    ) -> Result<(), String> {
        let slot =
            self.slots.get_mut(slot_id).ok_or_else(|| format!("slot {} not found", slot_id))?;

        slot.health.is_healthy = healthy;
        slot.health.latency_ms = latency_ms;
        slot.health.last_check_at = Some(Instant::now());

        if !healthy {
            slot.health.error_count += 1;
        }

        // Check rollback policy.
        if self.rollback_policy.auto_rollback {
            let should_rollback = slot.health.error_count >= self.rollback_policy.max_errors
                || latency_ms > self.rollback_policy.max_latency_ms;
            if should_rollback && slot.status == DeploymentStatus::Active {
                slot.status = DeploymentStatus::Failed;
            }
        }

        Ok(())
    }

    /// Run health checks on all non-pending slots with the given results.
    pub fn health_check_all(&mut self, results: &[(usize, bool, f64)]) -> Result<(), String> {
        for &(slot_id, healthy, latency_ms) in results {
            self.health_check(slot_id, healthy, latency_ms)?;
        }
        Ok(())
    }

    /// Check whether auto-rollback should trigger based on current health.
    pub fn should_auto_rollback(&self) -> bool {
        self.rollback_policy.auto_rollback
            && self.slots.iter().any(|s| {
                s.status == DeploymentStatus::Failed
                    || (s.status == DeploymentStatus::Active
                        && (s.health.error_count >= self.rollback_policy.max_errors
                            || s.health.latency_ms > self.rollback_policy.max_latency_ms))
            })
    }

    /// Mark a slot's deployment as failed.
    pub fn mark_failed(&mut self, slot_id: usize) -> Result<(), String> {
        let slot =
            self.slots.get_mut(slot_id).ok_or_else(|| format!("slot {} not found", slot_id))?;
        slot.status = DeploymentStatus::Failed;
        slot.traffic_weight = 0.0;
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ------------------------------------------------------------

    fn default_config() -> DeploymentConfig {
        DeploymentConfig::default()
    }

    fn make_model(id: &str, version: u64) -> ModelVersion {
        ModelVersion {
            id: id.to_string(),
            version,
            path: format!("/models/{id}/v{version}.gguf"),
            size_bytes: 1_000_000,
            created_at: 1_700_000_000 + version,
            status: ModelStatus::Available,
        }
    }

    fn manager() -> DeploymentManager {
        DeploymentManager::new(default_config()).unwrap()
    }

    fn deploy_active(mgr: &mut DeploymentManager, model: &ModelVersion) {
        mgr.deploy(model.clone(), RolloutStrategy::Immediate).unwrap();
        mgr.execute_all_steps().unwrap();
    }

    // -----------------------------------------------------------------------
    // Config validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default_is_valid() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn test_config_max_models_zero() {
        let mut c = default_config();
        c.max_models = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_health_check_interval_zero() {
        let mut c = default_config();
        c.health_check_interval_ms = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_canary_percentage_negative() {
        let mut c = default_config();
        c.canary_percentage = -1.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_canary_percentage_over_100() {
        let mut c = default_config();
        c.canary_percentage = 101.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_canary_percentage_boundaries() {
        let mut c = default_config();
        c.canary_percentage = 0.0;
        assert!(c.validate().is_ok());
        c.canary_percentage = 100.0;
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_manager_creation_with_invalid_config() {
        let mut c = default_config();
        c.max_models = 0;
        assert!(DeploymentManager::new(c).is_err());
    }

    // -----------------------------------------------------------------------
    // Slot initialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_slots_initialized_to_pending() {
        let mgr = manager();
        assert_eq!(mgr.slots().len(), 4);
        for slot in mgr.slots() {
            assert_eq!(slot.status, DeploymentStatus::Pending);
            assert!(slot.model_version.is_none());
            assert_eq!(slot.traffic_weight, 0.0);
        }
    }

    #[test]
    fn test_slot_ids_are_sequential() {
        let mgr = manager();
        for (i, slot) in mgr.slots().iter().enumerate() {
            assert_eq!(slot.slot_id, i);
        }
    }

    #[test]
    fn test_no_active_slot_initially() {
        let mgr = manager();
        assert!(mgr.get_active_slot().is_none());
    }

    #[test]
    fn test_no_serving_slots_initially() {
        let mgr = manager();
        assert!(mgr.get_serving_slots().is_empty());
    }

    // -----------------------------------------------------------------------
    // Deployment status
    // -----------------------------------------------------------------------

    #[test]
    fn test_deployment_status_terminal() {
        assert!(DeploymentStatus::Active.is_terminal());
        assert!(DeploymentStatus::RolledBack.is_terminal());
        assert!(DeploymentStatus::Failed.is_terminal());
        assert!(!DeploymentStatus::Pending.is_terminal());
        assert!(!DeploymentStatus::Deploying.is_terminal());
        assert!(!DeploymentStatus::Warming.is_terminal());
        assert!(!DeploymentStatus::Draining.is_terminal());
    }

    // -----------------------------------------------------------------------
    // Immediate deployment
    // -----------------------------------------------------------------------

    #[test]
    fn test_immediate_deploy_lifecycle() {
        let mut mgr = manager();
        let m = make_model("a", 1);
        let plan = mgr.deploy(m.clone(), RolloutStrategy::Immediate).unwrap();
        assert_eq!(plan.steps.len(), 2);
        assert!(mgr.active_plan().is_some());

        let steps = mgr.execute_all_steps().unwrap();
        assert_eq!(steps.len(), 2);
        assert!(mgr.active_plan().is_none());

        let active = mgr.get_active_slot().unwrap();
        assert_eq!(active.model_version.as_ref().unwrap().id, "a");
        assert_eq!(active.traffic_weight, 100.0);
    }

    #[test]
    fn test_immediate_deploy_sets_status_active() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        assert_eq!(mgr.get_active_slot().unwrap().status, DeploymentStatus::Active);
    }

    #[test]
    fn test_immediate_deploy_records_metrics() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        assert_eq!(mgr.metrics().total_deployments, 1);
        assert_eq!(mgr.metrics().successful_deployments, 1);
    }

    #[test]
    fn test_immediate_deploy_no_previous_version() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        assert!(mgr.version_history().is_empty());
    }

    #[test]
    fn test_immediate_deploy_stores_previous_version() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        deploy_active(&mut mgr, &make_model("a", 2));
        assert_eq!(mgr.version_history().len(), 1);
        assert_eq!(mgr.version_history()[0].version, 1);
    }

    // -----------------------------------------------------------------------
    // Blue/green deployment
    // -----------------------------------------------------------------------

    #[test]
    fn test_blue_green_plan_has_five_steps() {
        let mgr = manager();
        let plan = mgr.plan_deployment(make_model("a", 1), RolloutStrategy::BlueGreen).unwrap();
        assert_eq!(plan.steps.len(), 5);
    }

    #[test]
    fn test_blue_green_plan_steps_order() {
        let mgr = manager();
        let plan = mgr.plan_deployment(make_model("a", 1), RolloutStrategy::BlueGreen).unwrap();
        let actions: Vec<_> = plan.steps.iter().map(|s| s.action.clone()).collect();
        assert_eq!(
            actions,
            vec![
                DeploymentAction::LoadModel,
                DeploymentAction::Warmup,
                DeploymentAction::Validate,
                DeploymentAction::ShiftTraffic,
                DeploymentAction::DrainOld,
            ]
        );
    }

    #[test]
    fn test_blue_green_deploy_swaps_slots() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        let active_before = mgr.get_active_slot().unwrap().slot_id;

        mgr.deploy(make_model("a", 2), RolloutStrategy::BlueGreen).unwrap();
        mgr.execute_all_steps().unwrap();

        let active_after = mgr.get_active_slot().unwrap().slot_id;
        // Blue/green uses a different slot.
        assert_ne!(active_before, active_after);
    }

    #[test]
    fn test_blue_green_drains_old_slot() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));

        mgr.deploy(make_model("a", 2), RolloutStrategy::BlueGreen).unwrap();
        mgr.execute_all_steps().unwrap();

        assert_eq!(mgr.slots()[0].status, DeploymentStatus::Draining);
        assert_eq!(mgr.slots()[0].traffic_weight, 0.0);
    }

    #[test]
    fn test_blue_green_new_slot_gets_full_traffic() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.deploy(make_model("a", 2), RolloutStrategy::BlueGreen).unwrap();
        mgr.execute_all_steps().unwrap();
        let active = mgr.get_active_slot().unwrap();
        assert_eq!(active.traffic_weight, 100.0);
    }

    // -----------------------------------------------------------------------
    // Canary deployment
    // -----------------------------------------------------------------------

    #[test]
    fn test_canary_plan_has_gradual_steps() {
        let mgr = manager();
        let plan = mgr.plan_deployment(make_model("a", 1), RolloutStrategy::Canary(10.0)).unwrap();
        // Load + warmup + multiple shifts + drain
        assert!(plan.steps.len() >= 5);
    }

    #[test]
    fn test_canary_plan_ends_at_100_percent() {
        let mgr = manager();
        let plan = mgr.plan_deployment(make_model("a", 1), RolloutStrategy::Canary(10.0)).unwrap();
        let last_shift =
            plan.steps.iter().rfind(|s| s.action == DeploymentAction::ShiftTraffic).unwrap();
        assert_eq!(last_shift.traffic_shift, 100.0);
    }

    #[test]
    fn test_canary_plan_traffic_monotonically_increases() {
        let mgr = manager();
        let plan = mgr.plan_deployment(make_model("a", 1), RolloutStrategy::Canary(5.0)).unwrap();
        let shifts: Vec<f32> = plan
            .steps
            .iter()
            .filter(|s| s.action == DeploymentAction::ShiftTraffic)
            .map(|s| s.traffic_shift)
            .collect();
        for w in shifts.windows(2) {
            assert!(w[1] >= w[0], "traffic should not decrease: {} < {}", w[1], w[0]);
        }
    }

    #[test]
    fn test_canary_execute_full() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.deploy(make_model("a", 2), RolloutStrategy::Canary(25.0)).unwrap();
        mgr.execute_all_steps().unwrap();
        assert!(mgr.get_active_slot().is_some());
    }

    #[test]
    fn test_canary_shifts_require_validation() {
        let mgr = manager();
        let plan = mgr.plan_deployment(make_model("a", 1), RolloutStrategy::Canary(10.0)).unwrap();
        for step in &plan.steps {
            if step.action == DeploymentAction::ShiftTraffic {
                assert!(step.validation_required);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Rolling deployment
    // -----------------------------------------------------------------------

    #[test]
    fn test_rolling_plan_covers_all_slots() {
        let mgr = manager();
        let plan = mgr.plan_deployment(make_model("a", 1), RolloutStrategy::Rolling(1)).unwrap();
        let slot_ids: Vec<usize> = plan.steps.iter().map(|s| s.slot_id).collect();
        for i in 0..mgr.slots().len() {
            assert!(slot_ids.contains(&i), "slot {} not in plan", i);
        }
    }

    #[test]
    fn test_rolling_plan_batch_size_two() {
        let config = DeploymentConfig { max_models: 4, ..default_config() };
        let mgr = DeploymentManager::new(config).unwrap();
        let plan = mgr.plan_deployment(make_model("a", 1), RolloutStrategy::Rolling(2)).unwrap();
        // 4 slots / batch 2 = 2 batches; each slot gets 3 steps
        assert_eq!(plan.steps.len(), 12);
    }

    #[test]
    fn test_rolling_plan_batch_size_larger_than_slots() {
        let mgr = manager();
        let plan = mgr.plan_deployment(make_model("a", 1), RolloutStrategy::Rolling(10)).unwrap();
        // All slots in one batch.
        assert_eq!(plan.steps.len(), 4 * 3); // 3 steps per slot
    }

    #[test]
    fn test_rolling_deploy_execute() {
        let mut mgr = manager();
        mgr.deploy(make_model("a", 1), RolloutStrategy::Rolling(2)).unwrap();
        let steps = mgr.execute_all_steps().unwrap();
        assert!(!steps.is_empty());
    }

    // -----------------------------------------------------------------------
    // Rollback
    // -----------------------------------------------------------------------

    #[test]
    fn test_rollback_without_history_fails() {
        let mut mgr = manager();
        assert!(mgr.rollback().is_err());
    }

    #[test]
    fn test_rollback_restores_previous_version() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        deploy_active(&mut mgr, &make_model("a", 2));

        let prev = mgr.rollback().unwrap();
        assert_eq!(prev.version, 1);
    }

    #[test]
    fn test_rollback_sets_slot0_active() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        deploy_active(&mut mgr, &make_model("a", 2));
        mgr.rollback().unwrap();

        assert_eq!(mgr.slots()[0].status, DeploymentStatus::Active);
        assert_eq!(mgr.slots()[0].traffic_weight, 100.0);
    }

    #[test]
    fn test_rollback_cancels_active_plan() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.deploy(make_model("a", 2), RolloutStrategy::BlueGreen).unwrap();
        // Plan is active but we rollback before completing.
        mgr.rollback().unwrap();
        assert!(mgr.active_plan().is_none());
    }

    #[test]
    fn test_rollback_records_metrics() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        deploy_active(&mut mgr, &make_model("a", 2));
        mgr.rollback().unwrap();
        assert_eq!(mgr.metrics().rollbacks, 1);
    }

    #[test]
    fn test_double_rollback() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        deploy_active(&mut mgr, &make_model("a", 2));
        deploy_active(&mut mgr, &make_model("a", 3));

        mgr.rollback().unwrap();
        assert_eq!(mgr.slots()[0].model_version.as_ref().unwrap().version, 2);

        mgr.rollback().unwrap();
        assert_eq!(mgr.slots()[0].model_version.as_ref().unwrap().version, 1);
    }

    #[test]
    fn test_rollback_marks_other_slots_rolled_back() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        // Deploy via blue/green to occupy another slot.
        mgr.deploy(make_model("a", 2), RolloutStrategy::BlueGreen).unwrap();
        mgr.execute_all_steps().unwrap();
        mgr.rollback().unwrap();

        // Slot 1 was the blue/green target, should be rolled-back.
        assert_eq!(mgr.slots()[1].status, DeploymentStatus::RolledBack);
    }

    // -----------------------------------------------------------------------
    // Health checks
    // -----------------------------------------------------------------------

    #[test]
    fn test_health_check_updates_slot() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.health_check(0, true, 5.0).unwrap();
        assert!(mgr.slots()[0].health.is_healthy);
        assert_eq!(mgr.slots()[0].health.latency_ms, 5.0);
    }

    #[test]
    fn test_health_check_increments_error_count() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.health_check(0, false, 5.0).unwrap();
        mgr.health_check(0, false, 5.0).unwrap();
        assert_eq!(mgr.slots()[0].health.error_count, 2);
    }

    #[test]
    fn test_health_check_sets_last_check_at() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.health_check(0, true, 5.0).unwrap();
        assert!(mgr.slots()[0].health.last_check_at.is_some());
    }

    #[test]
    fn test_health_check_invalid_slot() {
        let mut mgr = manager();
        assert!(mgr.health_check(99, true, 5.0).is_err());
    }

    #[test]
    fn test_health_check_all() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.health_check_all(&[(0, true, 1.0), (1, true, 2.0)]).unwrap();
        assert_eq!(mgr.slots()[0].health.latency_ms, 1.0);
        assert_eq!(mgr.slots()[1].health.latency_ms, 2.0);
    }

    #[test]
    fn test_health_check_exceeds_max_errors_triggers_failed() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        for _ in 0..3 {
            mgr.health_check(0, false, 5.0).unwrap();
        }
        assert_eq!(mgr.slots()[0].status, DeploymentStatus::Failed);
    }

    #[test]
    fn test_health_check_high_latency_triggers_failed() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.health_check(0, true, 2000.0).unwrap();
        assert_eq!(mgr.slots()[0].status, DeploymentStatus::Failed);
    }

    #[test]
    fn test_should_auto_rollback_when_failed() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.mark_failed(0).unwrap();
        assert!(mgr.should_auto_rollback());
    }

    #[test]
    fn test_should_auto_rollback_disabled() {
        let policy = RollbackPolicy { auto_rollback: false, ..Default::default() };
        let mut mgr =
            DeploymentManager::new(default_config()).unwrap().with_rollback_policy(policy);
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.mark_failed(0).unwrap();
        assert!(!mgr.should_auto_rollback());
    }

    #[test]
    fn test_health_failure_does_not_trigger_rollback_when_disabled() {
        let policy = RollbackPolicy { auto_rollback: false, ..Default::default() };
        let mut mgr =
            DeploymentManager::new(default_config()).unwrap().with_rollback_policy(policy);
        deploy_active(&mut mgr, &make_model("a", 1));
        for _ in 0..5 {
            mgr.health_check(0, false, 5.0).unwrap();
        }
        // Status should remain Active when auto_rollback is off.
        assert_eq!(mgr.slots()[0].status, DeploymentStatus::Active);
    }

    // -----------------------------------------------------------------------
    // Draining
    // -----------------------------------------------------------------------

    #[test]
    fn test_drain_slot_sets_draining() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.drain_slot(0).unwrap();
        assert_eq!(mgr.slots()[0].status, DeploymentStatus::Draining);
        assert_eq!(mgr.slots()[0].traffic_weight, 0.0);
    }

    #[test]
    fn test_drain_slot_redistributes_traffic() {
        let mut mgr = manager();
        // Set up two active slots manually.
        mgr.slots[0].status = DeploymentStatus::Active;
        mgr.slots[0].traffic_weight = 50.0;
        mgr.slots[0].model_version = Some(make_model("a", 1));
        mgr.slots[1].status = DeploymentStatus::Active;
        mgr.slots[1].traffic_weight = 50.0;
        mgr.slots[1].model_version = Some(make_model("a", 1));

        mgr.drain_slot(0).unwrap();
        assert_eq!(mgr.slots()[1].traffic_weight, 100.0);
    }

    #[test]
    fn test_drain_invalid_slot() {
        let mut mgr = manager();
        assert!(mgr.drain_slot(99).is_err());
    }

    // -----------------------------------------------------------------------
    // Metrics
    // -----------------------------------------------------------------------

    #[test]
    fn test_metrics_initially_zero() {
        let mgr = manager();
        assert_eq!(mgr.metrics().total_deployments, 0);
        assert_eq!(mgr.metrics().successful_deployments, 0);
        assert_eq!(mgr.metrics().rollbacks, 0);
    }

    #[test]
    fn test_metrics_after_multiple_deployments() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        deploy_active(&mut mgr, &make_model("a", 2));
        assert_eq!(mgr.metrics().total_deployments, 2);
        assert_eq!(mgr.metrics().successful_deployments, 2);
    }

    #[test]
    fn test_metrics_avg_deploy_time() {
        let mut metrics = DeploymentMetrics::default();
        metrics.record_deployment(100.0, true);
        metrics.record_deployment(200.0, true);
        assert!((metrics.avg_deploy_time_ms - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_record_rollback() {
        let mut metrics = DeploymentMetrics::default();
        metrics.record_rollback();
        metrics.record_rollback();
        assert_eq!(metrics.rollbacks, 2);
    }

    #[test]
    fn test_metrics_uptime_tracking() {
        let mut metrics = DeploymentMetrics::default();
        metrics.add_uptime("v1", Duration::from_secs(60));
        metrics.add_uptime("v1", Duration::from_secs(40));
        metrics.add_uptime("v2", Duration::from_secs(30));
        assert_eq!(*metrics.uptime_per_version.get("v1").unwrap(), Duration::from_secs(100));
        assert_eq!(*metrics.uptime_per_version.get("v2").unwrap(), Duration::from_secs(30));
    }

    // -----------------------------------------------------------------------
    // Concurrent / multi-version management
    // -----------------------------------------------------------------------

    #[test]
    fn test_deploy_while_deploying_fails() {
        let mut mgr = manager();
        mgr.deploy(make_model("a", 1), RolloutStrategy::BlueGreen).unwrap();
        let err = mgr.deploy(make_model("a", 2), RolloutStrategy::Immediate);
        assert!(err.is_err());
    }

    #[test]
    fn test_sequential_deploys_different_models() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        deploy_active(&mut mgr, &make_model("b", 1));
        let active = mgr.get_active_slot().unwrap();
        assert_eq!(active.model_version.as_ref().unwrap().id, "b");
    }

    #[test]
    fn test_version_history_tracks_all_previous() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        deploy_active(&mut mgr, &make_model("a", 2));
        deploy_active(&mut mgr, &make_model("a", 3));
        assert_eq!(mgr.version_history().len(), 2);
        assert_eq!(mgr.version_history()[0].version, 1);
        assert_eq!(mgr.version_history()[1].version, 2);
    }

    // -----------------------------------------------------------------------
    // Execute step-by-step
    // -----------------------------------------------------------------------

    #[test]
    fn test_execute_next_step_returns_none_when_done() {
        let mut mgr = manager();
        mgr.deploy(make_model("a", 1), RolloutStrategy::Immediate).unwrap();
        mgr.execute_all_steps().unwrap();
        // No plan active anymore.
        assert!(mgr.execute_next_step().is_err());
    }

    #[test]
    fn test_execute_next_step_one_at_a_time() {
        let mut mgr = manager();
        mgr.deploy(make_model("a", 1), RolloutStrategy::Immediate).unwrap();
        let s1 = mgr.execute_next_step().unwrap().unwrap();
        assert_eq!(s1.action, DeploymentAction::LoadModel);
        let s2 = mgr.execute_next_step().unwrap().unwrap();
        assert_eq!(s2.action, DeploymentAction::ShiftTraffic);
    }

    // -----------------------------------------------------------------------
    // Model version / model status
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_version_equality() {
        let a = make_model("a", 1);
        let b = make_model("a", 1);
        assert_eq!(a, b);
    }

    #[test]
    fn test_model_version_inequality() {
        let a = make_model("a", 1);
        let b = make_model("a", 2);
        assert_ne!(a, b);
    }

    #[test]
    fn test_model_status_variants() {
        let statuses = [
            ModelStatus::Available,
            ModelStatus::Loading,
            ModelStatus::Loaded,
            ModelStatus::Unloading,
            ModelStatus::Failed,
        ];
        assert_eq!(statuses.len(), 5);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_slot_manager() {
        let config = DeploymentConfig { max_models: 1, ..default_config() };
        let mut mgr = DeploymentManager::new(config).unwrap();
        deploy_active(&mut mgr, &make_model("a", 1));
        assert_eq!(mgr.get_active_slot().unwrap().slot_id, 0);
    }

    #[test]
    fn test_rollback_policy_builder() {
        let policy = RollbackPolicy { max_errors: 10, max_latency_ms: 500.0, auto_rollback: true };
        let mgr = DeploymentManager::new(default_config()).unwrap().with_rollback_policy(policy);
        assert!(!mgr.should_auto_rollback());
    }

    #[test]
    fn test_mark_failed_clears_traffic() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.mark_failed(0).unwrap();
        assert_eq!(mgr.slots()[0].traffic_weight, 0.0);
        assert_eq!(mgr.slots()[0].status, DeploymentStatus::Failed);
    }

    #[test]
    fn test_mark_failed_invalid_slot() {
        let mut mgr = manager();
        assert!(mgr.mark_failed(99).is_err());
    }

    // -----------------------------------------------------------------------
    // Deployment plan inspection
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_records_source_version() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        let plan = mgr.plan_deployment(make_model("a", 2), RolloutStrategy::BlueGreen).unwrap();
        assert_eq!(plan.source_version.as_ref().unwrap().version, 1);
    }

    #[test]
    fn test_plan_records_target_version() {
        let mgr = manager();
        let plan = mgr.plan_deployment(make_model("a", 3), RolloutStrategy::Immediate).unwrap();
        assert_eq!(plan.target_version.version, 3);
    }

    #[test]
    fn test_plan_strategy_stored() {
        let mgr = manager();
        let plan = mgr.plan_deployment(make_model("a", 1), RolloutStrategy::Canary(15.0)).unwrap();
        assert_eq!(plan.strategy, RolloutStrategy::Canary(15.0));
    }

    // -----------------------------------------------------------------------
    // State machine property tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_traffic_weights_never_exceed_100() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.deploy(make_model("a", 2), RolloutStrategy::Canary(10.0)).unwrap();
        // Execute step by step, checking invariant.
        while mgr.active_plan().is_some() {
            mgr.execute_next_step().unwrap();
            let total: f32 = mgr.slots().iter().map(|s| s.traffic_weight).sum();
            assert!(total <= 100.1, "total traffic {total} > 100");
        }
    }

    #[test]
    fn test_at_most_one_active_plan() {
        let mut mgr = manager();
        mgr.deploy(make_model("a", 1), RolloutStrategy::Immediate).unwrap();
        let second = mgr.deploy(make_model("a", 2), RolloutStrategy::Immediate);
        assert!(second.is_err());
    }

    #[test]
    fn test_rollback_always_restores_to_active() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        deploy_active(&mut mgr, &make_model("a", 2));
        mgr.rollback().unwrap();
        let active = mgr.get_active_slot().unwrap();
        assert_eq!(active.status, DeploymentStatus::Active);
        assert!(active.model_version.is_some());
    }

    #[test]
    fn test_deploy_after_rollback_succeeds() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        deploy_active(&mut mgr, &make_model("a", 2));
        mgr.rollback().unwrap();
        // Should be able to deploy again.
        deploy_active(&mut mgr, &make_model("a", 3));
        assert_eq!(mgr.get_active_slot().unwrap().model_version.as_ref().unwrap().version, 3);
    }

    #[test]
    fn test_health_check_on_pending_slot_no_panic() {
        let mut mgr = manager();
        mgr.health_check(0, true, 1.0).unwrap();
        assert!(mgr.slots()[0].health.is_healthy);
    }

    #[test]
    fn test_drain_then_deploy_reuses_slot() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.drain_slot(0).unwrap();
        // The slot is draining; a new immediate deploy should still work on slot 0.
        deploy_active(&mut mgr, &make_model("a", 2));
        let active = mgr.get_active_slot().unwrap();
        assert!(active.model_version.is_some());
    }

    #[test]
    fn test_canary_with_50_percent() {
        let mgr = manager();
        let plan = mgr.plan_deployment(make_model("a", 1), RolloutStrategy::Canary(50.0)).unwrap();
        let shifts: Vec<f32> = plan
            .steps
            .iter()
            .filter(|s| s.action == DeploymentAction::ShiftTraffic)
            .map(|s| s.traffic_shift)
            .collect();
        assert!(!shifts.is_empty());
        assert_eq!(*shifts.last().unwrap(), 100.0);
    }

    #[test]
    fn test_immediate_plan_length() {
        let mgr = manager();
        let plan = mgr.plan_deployment(make_model("a", 1), RolloutStrategy::Immediate).unwrap();
        assert_eq!(plan.steps.len(), 2);
    }

    #[test]
    fn test_rolling_batch_zero_treated_as_one() {
        let mgr = manager();
        let plan = mgr.plan_deployment(make_model("a", 1), RolloutStrategy::Rolling(0)).unwrap();
        // batch_size.max(1) should prevent panic.
        assert!(!plan.steps.is_empty());
    }

    #[test]
    fn test_serving_slots_after_canary_partial() {
        let mut mgr = manager();
        deploy_active(&mut mgr, &make_model("a", 1));
        mgr.deploy(make_model("a", 2), RolloutStrategy::Canary(10.0)).unwrap();
        // Execute load + warmup + first shift.
        mgr.execute_next_step().unwrap(); // load
        mgr.execute_next_step().unwrap(); // warmup
        mgr.execute_next_step().unwrap(); // first shift (10%)
        let serving = mgr.get_serving_slots();
        // At least the canary slot should be serving.
        assert!(!serving.is_empty());
    }

    #[test]
    fn test_default_health_check() {
        let hc = HealthCheck::default();
        assert!(hc.is_healthy);
        assert_eq!(hc.latency_ms, 0.0);
        assert_eq!(hc.error_count, 0);
        assert!(hc.last_check_at.is_none());
    }

    #[test]
    fn test_default_rollback_policy() {
        let rp = RollbackPolicy::default();
        assert_eq!(rp.max_errors, 3);
        assert_eq!(rp.max_latency_ms, 1000.0);
        assert!(rp.auto_rollback);
    }

    #[test]
    fn test_deployment_action_eq() {
        assert_eq!(DeploymentAction::LoadModel, DeploymentAction::LoadModel);
        assert_ne!(DeploymentAction::LoadModel, DeploymentAction::Warmup);
    }
}
