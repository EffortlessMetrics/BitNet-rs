//! Core startup contract diagnostics shared by wrappers and runtime frontends.

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
        self.info.extend(report_lines::info_lines(self));
        self.warnings.extend(report_lines::warning_lines(&self.contract));
    }
}

mod report_lines {
    use super::{ProfileContract, StartupContractReport};

    pub(super) fn info_lines(report: &StartupContractReport) -> [String; 2] {
        [report.contract.summary(), format!("Profile summary: {}", report.profile_summary())]
    }

    pub(super) fn warning_lines(contract: &ProfileContract) -> Vec<String> {
        let mut warnings = Vec::new();

        if !contract.is_compatible() {
            warnings.push(format!(
                "Startup contract is non-compliant: missing={:?} forbidden={:?}",
                contract.missing_required(),
                contract.forbidden_active()
            ));
        }

        if has_profile_violations(contract) {
            warnings.push(format!(
                "Profile violations for active build: missing={:?} forbidden={:?}",
                contract.missing_required(),
                contract.forbidden_active()
            ));
        }

        warnings
    }

    fn has_profile_violations(contract: &ProfileContract) -> bool {
        !contract.missing_required().is_empty() || !contract.forbidden_active().is_empty()
    }
}

fn join_features(features: &[String]) -> String {
    if features.is_empty() { "none".to_string() } else { features.join("+") }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPONENTS: [RuntimeComponent; 4] = [
        RuntimeComponent::Cli,
        RuntimeComponent::Server,
        RuntimeComponent::Test,
        RuntimeComponent::Custom,
    ];

    #[test]
    fn join_features_returns_none_for_empty_slice() {
        let features: Vec<String> = Vec::new();

        assert_eq!(join_features(&features), "none");
    }

    #[test]
    fn join_features_preserves_order_with_plus_delimiters() {
        let features = vec!["cpu".to_string(), "tokenizers".to_string(), "cli".to_string()];

        assert_eq!(join_features(&features), "cpu+tokenizers+cli");
    }

    #[test]
    fn evaluate_observe_builds_report_for_each_component() -> Result<()> {
        for component in COMPONENTS {
            let report = StartupContractReport::evaluate(component, ContractPolicy::Observe)?;

            assert_eq!(report.contract.component().label(), component.label());
            assert!(!report.info.is_empty(), "report must include informational diagnostics");
            assert!(
                report.info.iter().any(|line| line.contains(component.label())),
                "diagnostics should mention component label {}",
                component.label()
            );
        }
        Ok(())
    }

    #[test]
    fn profile_summary_includes_context_and_feature_sections() -> Result<()> {
        let report =
            StartupContractReport::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe)?;
        let summary = report.profile_summary();

        assert!(summary.contains("scenario="));
        assert!(summary.contains("/environment="));
        assert!(summary.contains(",required="));
        assert!(summary.contains(",optional="));
        assert!(summary.contains(",forbidden="));
        Ok(())
    }

    #[test]
    fn populate_lines_always_adds_summary_and_profile_summary() -> Result<()> {
        let report =
            StartupContractReport::evaluate(RuntimeComponent::Test, ContractPolicy::Observe)?;

        assert_eq!(report.info.len(), 2);
        assert_eq!(report.info[0], report.contract.summary());
        assert_eq!(report.info[1], format!("Profile summary: {}", report.profile_summary()));
        Ok(())
    }

    #[test]
    fn warnings_reflect_contract_compatibility() -> Result<()> {
        let report =
            StartupContractReport::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe)?;

        if report.contract.is_compatible() {
            assert!(report.warnings.is_empty());
        } else {
            assert!(
                report
                    .warnings
                    .iter()
                    .any(|line| line.contains("Startup contract is non-compliant"))
            );
        }
        Ok(())
    }

    #[test]
    fn violation_warning_is_present_only_when_violation_lists_are_non_empty() -> Result<()> {
        let report =
            StartupContractReport::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe)?;
        let has_violations = !report.contract.missing_required().is_empty()
            || !report.contract.forbidden_active().is_empty();
        let has_violation_warning =
            report.warnings.iter().any(|line| line.contains("Profile violations for active build"));

        assert_eq!(has_violation_warning, has_violations);
        Ok(())
    }

    #[test]
    fn report_keeps_contract_feature_lists_in_profile_summary() -> Result<()> {
        let report =
            StartupContractReport::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe)?;
        let summary = report.profile_summary();

        assert!(
            summary.contains(&format!(
                "required={}",
                join_features(report.contract.required_features())
            ))
        );
        assert!(
            summary.contains(&format!(
                "optional={}",
                join_features(report.contract.optional_features())
            ))
        );
        assert!(summary.contains(&format!(
            "forbidden={}",
            join_features(report.contract.forbidden_features())
        )));
        Ok(())
    }
}
