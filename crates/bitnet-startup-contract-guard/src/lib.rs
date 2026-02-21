//! Runtime startup contract orchestration for apps built on BitNet.
//!
//! This microcrate provides a single helper for evaluating startup contracts,
//! collecting BDD/profile diagnostics, and emitting a consistent log surface.

use anyhow::Result;
use tracing::{info, warn};

pub use bitnet_runtime_bootstrap::{
    ContractPolicy, RuntimeComponent, active_profile_summary, active_profile_violation_labels,
    feature_line,
};
pub use bitnet_startup_contract_diagnostics::StartupContractReport;

/// Compact snapshot of startup contract evaluation results.
#[derive(Debug)]
pub struct StartupContractGuard {
    /// Target runtime component being checked.
    pub component: RuntimeComponent,
    /// Enforcement policy used for evaluation.
    pub policy: ContractPolicy,
    /// Snapshot of active feature flags in canonical format.
    pub feature_line: String,
    /// Snapshot of selected active BDD profile row summary.
    pub profile_summary: String,
    /// Snapshot of active feature violations for the selected BDD cell, if any.
    pub profile_violations: Option<(Vec<String>, Vec<String>)>,
    /// Full startup contract report for downstream consumers.
    pub report: StartupContractReport,
}

impl StartupContractGuard {
    /// Evaluate the startup contract and return a reusable snapshot.
    pub fn evaluate(component: RuntimeComponent, policy: ContractPolicy) -> Result<Self> {
        let report = StartupContractReport::evaluate(component, policy)?;
        Ok(Self {
            component,
            policy,
            feature_line: feature_line(),
            profile_summary: active_profile_summary(),
            profile_violations: active_profile_violation_labels(),
            report,
        })
    }

    /// Check if the active profile is compatible after evaluation.
    pub fn is_compatible(&self) -> bool {
        self.report.contract.is_compatible()
    }

    /// Emit the standard contract diagnostics through tracing.
    pub fn emit_to_tracing(&self) {
        for line in self.report.info.iter() {
            info!(component = ?self.component, policy = ?self.policy, "{}", line);
        }
        for line in self.report.warnings.iter() {
            warn!(component = ?self.component, policy = ?self.policy, "{}", line);
        }

        match &self.profile_violations {
            Some((missing, forbidden)) if !missing.is_empty() || !forbidden.is_empty() => {
                warn!(
                    component = ?self.component,
                    missing = ?missing,
                    forbidden = ?forbidden,
                    "profile_violation=active"
                );
            }
            _ => {}
        }
    }
}

/// Evaluate and emit startup diagnostics in one call.
pub fn evaluate_and_emit(
    component: RuntimeComponent,
    policy: ContractPolicy,
) -> Result<StartupContractGuard> {
    let snapshot = StartupContractGuard::evaluate(component, policy)?;
    snapshot.emit_to_tracing();
    Ok(snapshot)
}
