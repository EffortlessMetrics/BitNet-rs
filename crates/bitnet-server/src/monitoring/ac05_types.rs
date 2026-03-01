//! AC05 health check data structures.
//!
//! This module intentionally re-exports the shared types from
//! `bitnet-server-health-types-core` to keep existing import paths stable.

pub use bitnet_server_health_types_core::{
    Ac05HealthResponse, LivenessResponse, PerformanceIndicators, ReadinessChecks,
    ReadinessResponse, SystemMetrics,
};
