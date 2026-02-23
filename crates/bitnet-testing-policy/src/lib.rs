//! Compatibility façade over `bitnet-testing-policy-core`.
//!
//! This crate keeps the existing `bitnet-testing-policy` API stable for downstream
//! consumers while the orchestration implementation lives in the dedicated
//! microcrate.

#![deny(unused_must_use)]

pub use bitnet_testing_policy_core::*;
