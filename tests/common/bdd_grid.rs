//! Compatibility façade over `bitnet_testing_policy_tests` for legacy `tests`
//! imports.
//!
//! This keeps existing `tests/common` API paths stable while moving the core
//! BDD grid + feature-flag contract logic into a focused, reusable crate.

pub use bitnet_testing_policy_tests::{
    ActiveContext, ActiveProfile, BddCell, BddGrid, BitnetFeature, ConfigurationContext,
    EnvironmentType, ExecutionEnvironment, FeatureSet, GridEnvironment, GridScenario,
    PolicyContract, PolicyDiagnostics, ScenarioConfigManager, TestConfigProfile, active_features,
    active_profile, active_profile_for, active_profile_for_context, active_profile_summary,
    active_profile_violation_labels, active_runtime_features, canonical_grid,
    context_from_environment, diagnostics, diagnostics_for_context, drift_check,
    drift_check as runtime_drift_check, feature_contract_snapshot, feature_labels, feature_line,
    from_grid_environment, from_grid_scenario, runtime_feature_labels, runtime_feature_line,
    to_grid_environment, to_grid_scenario, validate_active_profile,
    validate_active_profile_from_context, validate_explicit_profile, validate_profile_for_context,
};
