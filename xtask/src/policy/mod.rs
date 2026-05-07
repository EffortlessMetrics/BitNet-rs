//! BitNet-rs policy checkers (CI lane whitelist, file policy, no-panic, clippy
//! exceptions, lint inheritance).
//!
//! Each submodule implements a single check that the `xtask` CLI exposes
//! through `policy <subcommand>` (or its dedicated alias).
//!
//! Many of the structs in this module use `#[derive(Deserialize)]` for fields
//! that are validated as schema but not necessarily read from Rust — those
//! fields exist so TOML parsing fails on missing data, even though only some
//! are used at runtime. Dead-code warnings on those fields are suppressed.

#![allow(dead_code)]

pub mod ci_lanes;
pub mod clippy;
pub mod file_policy;
pub mod lints;
pub mod no_panic;
