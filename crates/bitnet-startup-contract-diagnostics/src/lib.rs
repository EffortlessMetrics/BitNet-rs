//! Shared startup contract diagnostics used by binaries and services.
//!
//! This crate centralizes runtime profile validation messaging for anything
//! launching a BitNet process, so diagnostics stay consistent across CLI,
//! server, and future runtime frontends.

use anyhow::Result;
pub use bitnet_runtime_bootstrap::{
    ContractPolicy, ContractState, ProfileContract, RuntimeComponent,
};

/// Result package for startup contract inspection and diagnostics.
#[derive(Debug)]
pub struct StartupContractReport {
    /// Evaluated startup contract.
    pub contract: ProfileContract,
    /// Informational messages safe for standard logging.
    pub info: Vec<String>,
    /// Warning messages, e.g. compatibility or feature mismatches.
    pub warnings: Vec<String>,
}

impl StartupContractReport {
    /// Build a report for a runtime component with an explicit policy.
    pub fn evaluate(component: RuntimeComponent, policy: ContractPolicy) -> Result<Self> {
        let contract = ProfileContract::evaluate(component, policy).enforce()?;
        let mut report = Self { contract, info: Vec::new(), warnings: Vec::new() };
        report.populate_lines();
        Ok(report)
    }

    /// Human-readable profile summary for the active BDD row.
    pub fn profile_summary(&self) -> String {
        let context = self.contract.context();
        let required = join_features(self.contract.required_features());
        let optional = join_features(self.contract.optional_features());
        let forbidden = join_features(self.contract.forbidden_features());
        format!(
            "scenario={}/environment={},required={},optional={},forbidden={}",
            context.scenario, context.environment, required, optional, forbidden
        )
    }

    fn populate_lines(&mut self) {
        self.info.push(self.contract.summary());
        self.info.push(format!("Profile summary: {}", self.profile_summary()));

        if !self.contract.is_compatible() {
            self.warnings.push(format!(
                "Startup contract is non-compliant: missing={:?} forbidden={:?}",
                self.contract.missing_required(),
                self.contract.forbidden_active()
            ));
        }

        if !self.contract.missing_required().is_empty()
            || !self.contract.forbidden_active().is_empty()
        {
            self.warnings.push(format!(
                "Profile violations for active build: missing={:?} forbidden={:?}",
                self.contract.missing_required(),
                self.contract.forbidden_active()
            ));
        }
    }
}

fn join_features(features: &[String]) -> String {
    if features.is_empty() { "none".to_string() } else { features.join("+") }
}
