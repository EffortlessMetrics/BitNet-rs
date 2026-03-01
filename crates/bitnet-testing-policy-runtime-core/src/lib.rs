//! Core runtime policy-state model and resolvers.

#![deny(unused_must_use)]

pub use bitnet_testing_policy_interop::{
    ActiveContext, ActiveProfile, ConfigurationContext, ExecutionEnvironment,
    FeatureContractSnapshot, FeatureSet, ScenarioConfigManager, TestConfigProfile, active_profile,
    active_profile_for, active_profile_summary, drift_check, feature_contract_snapshot,
    validate_profile_for_context,
};

pub use bitnet_testing_policy_state_core::{
    Environment, RuntimePolicyState, context_from_environment, detect_runtime_state,
    resolve_runtime_profile,
};
