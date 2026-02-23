#![deny(unused_must_use)]

//! Compatibility façade over `bitnet-testing-scenarios-core`.
//!
//! This crate preserves the historical crate API while keeping concrete scenario
//! model and resolver implementations in a dedicated microcrate.

pub use bitnet_testing_scenarios_core::*;
